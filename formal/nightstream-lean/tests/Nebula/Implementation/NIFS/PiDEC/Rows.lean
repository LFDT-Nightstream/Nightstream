import Nightstream.Implementation.Nebula.NIFS.PiDEC.TypedBridge

/-! Regression surface for the exact V2 product-PiDEC row bridge. -/

set_option autoImplicit false

namespace tests.NebulaProductPiDecRows

#check Nightstream.Implementation.Nebula.ProductPiDecRows.rows_length
#check Nightstream.Implementation.Nebula.ProductPiDecRows.rows_sound
#check Nightstream.Implementation.Nebula.ProductPiDecTypedBridge.typedEquations_of_rows
#check Nightstream.Implementation.Nebula.ProductPiDecTypedBridge.paperAccepted_of_rows_for_attempt
#check Nightstream.Implementation.Nebula.ProductPiDecTypedBridge.paperAccepted_of_rows
#check Nightstream.Implementation.Nebula.ProductPiDecTypedBridge.piDecCheck_true_of_rows

example (layout :
    Nightstream.Implementation.Nebula.ProductPiDecRows.Layout) :
    (Nightstream.Implementation.Nebula.ProductPiDecRows.rows layout).length =
      5400 :=
  Nightstream.Implementation.Nebula.ProductPiDecRows.rows_length layout

end tests.NebulaProductPiDecRows
