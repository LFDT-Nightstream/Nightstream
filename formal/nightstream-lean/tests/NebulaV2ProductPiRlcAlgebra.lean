import Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSound

/-! Regression surface for the exact aggregate V2 PiRLC algebra rows. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductPiRlcAlgebra

#check Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows.familyOrdinal_injective
#check Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows.family_windows_disjoint
#check Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows.family_satisfies
#check Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSound.publicBlock_publicOfRings
#check Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSound.bundleEquation_of_rows
#check Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSound.publicEquation_of_rows
#check Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSound.evaluationEquation_of_rows
#check Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSound.typedEquations_of_rows

example :
    Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows.families.length =
      110 :=
  Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows.families_length

example (layout :
    Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows.Layout) :
    (Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows.rows
      layout).length = 4817340 :=
  Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows.rows_length layout

example (layout :
    Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows.Layout) :
    (Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows.allocation
      layout).length = 4811400 :=
  Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows.allocation_length
    layout

end tests.NebulaV2ProductPiRlcAlgebra
