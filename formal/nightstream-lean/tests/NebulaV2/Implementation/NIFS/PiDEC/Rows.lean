import Nightstream.Implementation.NebulaV2.NIFS.PiDEC.TypedBridge

/-! Regression surface for the exact V2 product-PiDEC row bridge. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductPiDecRows

#check Nightstream.Implementation.NebulaV2.ProductPiDecRows.rows_length
#check Nightstream.Implementation.NebulaV2.ProductPiDecRows.rows_sound
#check Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge.typedEquations_of_rows
#check Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge.paperAccepted_of_rows_for_attempt
#check Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge.paperAccepted_of_rows
#check Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge.piDecCheck_true_of_rows

example (layout :
    Nightstream.Implementation.NebulaV2.ProductPiDecRows.Layout) :
    (Nightstream.Implementation.NebulaV2.ProductPiDecRows.rows layout).length =
      5400 :=
  Nightstream.Implementation.NebulaV2.ProductPiDecRows.rows_length layout

end tests.NebulaV2ProductPiDecRows
