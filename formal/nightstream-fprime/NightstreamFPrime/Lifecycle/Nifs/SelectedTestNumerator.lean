import NightstreamFPrime.Lifecycle.Types

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.SelectedTestNumerator

def width : Nat := 8

/-- SuperNeo v1.2 Appendix B.2, equation (14). -/
def numerator : Nat :=
  productionShape.cubeVariables * width +
    (productionShape.jointCoefficientCount - 1 + productionShape.cubeVariables)

theorem numerator_eq : numerator = 4589 := by
  rfl

end NightstreamFPrime.Lifecycle.Nifs.SelectedTestNumerator
