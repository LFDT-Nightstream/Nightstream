import NightstreamFPrime.Layout.Stage1.Spartan

/-! Default-profile values for the derived Spartan column boundaries. Core
map proofs use count relationships and do not import this module. -/

namespace NightstreamFPrime.Layout.Stage1.Spartan

theorem appendedPrivateColumnCount_eq :
    appendedPrivateColumnCount = 11045747 := by
  rfl

theorem sourceColumnCount_eq : SourceColumnCount = 20171875 := by
  rfl

theorem privateColumnCount_eq : privateColumnCount = 20171597 := by
  rfl

theorem constantColumn_eq : constantColumn = 20171597 := by
  exact privateColumnCount_eq

theorem spartanColumnCount_eq : spartanColumnCount = 20171876 := by
  rfl

theorem privateColumnCount_bound : privateColumnCount ≤ domainSize := by
  rw [privateColumnCount_eq, domainSize_eq]
  norm_num

attribute [simp] appendedPrivateColumnCount_eq sourceColumnCount_eq
  privateColumnCount_eq constantColumn_eq spartanColumnCount_eq
attribute [simp] pilotPrivateColumnCount pilotSourceColumnCount
  pilotPublicColumnCount expectedContextColumnCount

end NightstreamFPrime.Layout.Stage1.Spartan
