import Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationSound

/-! Regression surface for one exact V2 PiRLC ring-combination occurrence. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductPiRlcRingCombination

#check Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationRows.rows_sound
#check Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationSound.sourceOutputTerms_field
#check Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationSound.rows_imply_ring_combination

example :
    Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationRows.productCount =
      43740 :=
  Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationRows.productCount_eq

example (layout :
    Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationRows.Layout) :
    (Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationRows.rows
      layout).length = 43794 :=
  Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationRows.rows_length
    layout

example : 110 * 43794 = 4817340 := by decide

end tests.NebulaV2ProductPiRlcRingCombination
