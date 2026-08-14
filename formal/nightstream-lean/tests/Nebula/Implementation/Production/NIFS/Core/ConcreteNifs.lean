import Nightstream.Implementation.Nebula.Production.NIFS.Core.ConcreteNifs

/-! Regression surface for the production-profile executable paper-NIFS key. -/

set_option autoImplicit false

namespace tests.NebulaProductionProductConcreteNifs

open Nightstream.Implementation.Nebula.ProductionProductConcreteNifsKey
open Nightstream.Implementation.Nebula.ProductionProductConcreteNifs

#check publicAbsorber
#check SelectedKey
#check selectedKey
#check SelectedKey.publicInputState_eq
#check selectedKey_publicInputState
#check rows_imply_selectedKey_publicInputState

end tests.NebulaProductionProductConcreteNifs
