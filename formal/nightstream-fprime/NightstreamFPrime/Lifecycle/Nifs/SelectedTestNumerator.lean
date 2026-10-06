import NightstreamFPrime.Lifecycle.Types

/-!
Owns the selected PiCCS test-error numerator over `q²` as a natural number,
for export to Rust. `VerifierErrorBudget.test_error_eq_selected` proves that
it is the numerator of `IndependentExecution.testError` for the selected
shape and width.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.SelectedTestNumerator

/-- Selected sum-check message width, equal to `ProductionKey.degreeBound`. -/
def width : Nat := 9

/-- SuperNeo v1.2 equation (16): `log m · width` from the sum-check plus
`k·d·(t+1) + 2K + k - 1 + log m` from the joint gamma and alpha mixing. -/
def numerator : Nat :=
  productionShape.cubeVariables * width +
    (productionShape.jointCoefficientCount - 1 + productionShape.cubeVariables)

theorem numerator_eq : numerator = 7209 := by
  rfl

end NightstreamFPrime.Lifecycle.Nifs.SelectedTestNumerator
