import Nightstream.Implementation.NebulaV2.Production.NIFS.Core.ConcreteNifs

/-! Regression surface for the production-profile executable paper-NIFS key. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionProductConcreteNifs

open Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifsKey
open Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifs

#check publicAbsorber
#check SelectedKey
#check selectedKey
#check SelectedKey.publicInputState_eq
#check selectedKey_publicInputState
#check rows_imply_selectedKey_publicInputState

end tests.NebulaV2ProductionProductConcreteNifs
