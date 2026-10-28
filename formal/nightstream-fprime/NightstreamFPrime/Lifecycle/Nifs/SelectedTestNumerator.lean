import NightstreamFPrime.Lifecycle.Types

/-!
Owns the selected PiCCS test-error numerator over `q²` as a natural number,
for export to Rust.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.SelectedTestNumerator

def width : Nat := 8

/-- SuperNeo v1.2 Appendix B.2 equation (14): `log m · width` from the sum-check plus
`k·d·(t+1) + 2K + k - 1 + log m` from the joint gamma and alpha mixing. -/
def numerator : Nat :=
  productionShape.cubeVariables * width +
    (productionShape.jointCoefficientCount - 1 + productionShape.cubeVariables)

theorem numerator_eq : numerator = 4589 := by
  rfl

end NightstreamFPrime.Lifecycle.Nifs.SelectedTestNumerator
