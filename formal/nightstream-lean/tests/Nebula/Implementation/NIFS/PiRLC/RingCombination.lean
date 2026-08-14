import Nightstream.Implementation.Nebula.NIFS.PiRLC.RingCombinationSound

/-! Regression surface for one exact V2 PiRLC ring-combination occurrence. -/

set_option autoImplicit false

namespace tests.NebulaProductPiRlcRingCombination

#check Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows.rows_sound
#check Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound.sourceOutputTerms_field
#check Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound.rows_imply_ring_combination

example :
    Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows.productCount =
      43740 :=
  Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows.productCount_eq

example (layout :
    Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows.Layout) :
    (Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows.rows
      layout).length = 43794 :=
  Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows.rows_length
    layout

example : 110 * 43794 = 4817340 := by decide

end tests.NebulaProductPiRlcRingCombination
