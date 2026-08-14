import Nightstream.Implementation.Nebula.Production.Memory.BatchCarrierBridge

/-! Regression surface for the production full-claim memory carrier bridge. -/

set_option autoImplicit false

namespace tests.NebulaProductionMemoryBatchCarrierBridge

open Nightstream.Implementation.Nebula.ProductionMemoryBatchCarrierBridge

#check Placement
#check ofFn_claimAt
#check claim_eq_claimAt
#check suffixBatch_eq
#check rows_bind_and_consume_full_claim_memory

end tests.NebulaProductionMemoryBatchCarrierBridge
