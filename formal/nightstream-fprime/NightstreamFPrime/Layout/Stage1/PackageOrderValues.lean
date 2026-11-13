import NightstreamFPrime.Layout.Stage1.PackageOrder

namespace NightstreamFPrime.Layout.Stage1.PackageOrder

theorem appendedPrivateColumnCount_eq :
    appendedPrivateColumnCount = 6071918 := by
  rfl

theorem sourceColumnCount_eq : SourceColumnCount = 11415252 := by
  rfl

theorem privateColumnCount_eq : privateColumnCount = 11414924 := by
  rfl

theorem constantColumn_eq : constantColumn = 11414924 := by
  exact privateColumnCount_eq

theorem packageColumnCount_eq : packageColumnCount = 11415253 := by
  rfl

theorem privateColumnCount_bound : privateColumnCount ≤ domainSize := by
  rw [privateColumnCount_eq, domainSize_eq]
  norm_num

attribute [simp] appendedPrivateColumnCount_eq sourceColumnCount_eq
  privateColumnCount_eq constantColumn_eq packageColumnCount_eq
attribute [simp] pilotPrivateColumnCount pilotSourceColumnCount
  pilotPublicColumnCount expectedContextColumnCount

end NightstreamFPrime.Layout.Stage1.PackageOrder
