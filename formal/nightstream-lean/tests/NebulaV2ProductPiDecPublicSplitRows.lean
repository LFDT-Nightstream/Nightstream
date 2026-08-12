import Nightstream.Implementation.NebulaV2.ProductPiDecPublicSplitRows

/-! Regression surface for the exact V2 PiDEC public-input split rows. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductPiDecPublicSplitRows

open Nightstream.Implementation.NebulaV2.ProductPiDecPublicSplitRows

#check rows_length
#check rows_sound

example (layout : Layout) : (rows layout).length = 23760 :=
  rows_length layout

end tests.NebulaV2ProductPiDecPublicSplitRows
