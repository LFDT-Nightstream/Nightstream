import Nightstream.Implementation.Nebula.NIFS.PiDEC.PublicSplitRows

/-! Regression surface for the exact V2 PiDEC public-input split rows. -/

set_option autoImplicit false

namespace tests.NebulaProductPiDecPublicSplitRows

open Nightstream.Implementation.Nebula.ProductPiDecPublicSplitRows

#check rows_length
#check rows_sound

example (layout : Layout) : (rows layout).length = 23760 :=
  rows_length layout

end tests.NebulaProductPiDecPublicSplitRows
