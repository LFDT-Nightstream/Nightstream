import Nightstream.Implementation.NebulaV2.Production.Memory.ProductAccumulation

/-! Regression surface for exact row-derived product accumulation. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionMemoryProductAccumulation

open Nightstream.Implementation.NebulaV2.ProductionMemoryProductAccumulation

#check Run.activeWellFormed
#check Run.accumulatedProductsBalanced
#check Run.accumulatedFromConcreteOneBalanced

end tests.NebulaV2ProductionMemoryProductAccumulation
