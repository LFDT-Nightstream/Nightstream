import Nightstream.Implementation.Nebula.NIFS.PiRLC.AlgebraSound

/-! Regression surface for the exact aggregate V2 PiRLC algebra rows. -/

set_option autoImplicit false

namespace tests.NebulaProductPiRlcAlgebra

#check Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.familyOrdinal_injective
#check Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.family_windows_disjoint
#check Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.family_satisfies
#check Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSound.publicBlock_publicOfRings
#check Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSound.bundleEquation_of_rows
#check Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSound.publicEquation_of_rows
#check Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSound.evaluationEquation_of_rows
#check Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSound.typedEquations_of_rows

example :
    Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.families.length =
      110 :=
  Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.families_length

example (layout :
    Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.Layout) :
    (Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.rows
      layout).length = 4817340 :=
  Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.rows_length layout

example (layout :
    Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.Layout) :
    (Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.allocation
      layout).length = 4811400 :=
  Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.allocation_length
    layout

end tests.NebulaProductPiRlcAlgebra
