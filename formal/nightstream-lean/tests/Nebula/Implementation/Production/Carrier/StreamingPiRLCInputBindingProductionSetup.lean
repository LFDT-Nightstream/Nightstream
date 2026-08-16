import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBindingProductionSetup

/-! Regression surface for the fixed production PiRLC input setup. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcInputBindingProductionSetup

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingProductionSetup

#check rustSeedBytes
#check exact_rust_identity
#check exact_chunk_geometry
#check productionSetup
#check exact_identity

example : productionSetup.setup.seed.bytes = List.replicate 32 201 := by
  rfl

example : productionSetup.setup.rejectionFuel = 16 := by
  rfl

end tests.NebulaProductionStreamingPiRlcInputBindingProductionSetup
