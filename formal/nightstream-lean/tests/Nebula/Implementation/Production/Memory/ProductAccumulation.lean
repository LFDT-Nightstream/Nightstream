import Nightstream.Implementation.Nebula.Production.Memory.ProductAccumulation

/-! Regression surface for exact row-derived product accumulation. -/

set_option autoImplicit false

namespace tests.NebulaProductionMemoryProductAccumulation

open Nightstream.Implementation.Nebula.ProductionMemoryProductAccumulation

#check Run.activeWellFormed
#check Run.accumulatedProductsBalanced
#check Run.accumulatedFromConcreteOneBalanced

end tests.NebulaProductionMemoryProductAccumulation
