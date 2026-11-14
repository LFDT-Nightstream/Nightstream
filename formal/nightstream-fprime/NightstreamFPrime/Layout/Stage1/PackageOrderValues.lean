import NightstreamFPrime.Layout.Stage1.PackageOrder

namespace NightstreamFPrime.Layout.Stage1.PackageOrder

theorem appendedPrivateColumnCount_eq :
    appendedPrivateColumnCount = 5426736 := by
  rfl

theorem sourceColumnCount_eq : SourceColumnCount = 10434618 := by
  rfl

theorem privateColumnCount_eq : privateColumnCount = 10434290 := by
  rfl

theorem constantColumn_eq : constantColumn = 10434290 := by
  exact privateColumnCount_eq

theorem packageColumnCount_eq : packageColumnCount = 10434619 := by
  rfl

theorem privateColumnCount_bound : privateColumnCount ≤ domainSize := by
  rw [privateColumnCount_eq, domainSize_eq]
  norm_num

attribute [simp] appendedPrivateColumnCount_eq sourceColumnCount_eq
  privateColumnCount_eq constantColumn_eq packageColumnCount_eq
attribute [simp] pilotPrivateColumnCount pilotSourceColumnCount
  pilotPublicColumnCount expectedContextColumnCount

end NightstreamFPrime.Layout.Stage1.PackageOrder
