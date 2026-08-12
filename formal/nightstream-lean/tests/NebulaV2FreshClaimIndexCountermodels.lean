import Nightstream.Implementation.NebulaV2.FreshClaimIndexCountermodels

/-! Regression surface for the F-prime fresh-claim index countermodels. -/

set_option autoImplicit false

namespace tests.NebulaV2FreshClaimIndexCountermodels

open Nightstream.Implementation.NebulaV2.FreshClaimIndexCountermodels

#check consumed_bridge_does_not_imply_produced_bridge
#check consumed_bridge_rejects_valid_successor

end tests.NebulaV2FreshClaimIndexCountermodels
