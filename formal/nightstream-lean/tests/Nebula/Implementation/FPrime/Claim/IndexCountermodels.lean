import Nightstream.Implementation.Nebula.FPrime.Claim.IndexCountermodels

/-! Regression surface for the F-prime fresh-claim index countermodels. -/

set_option autoImplicit false

namespace tests.NebulaFreshClaimIndexCountermodels

open Nightstream.Implementation.Nebula.FreshClaimIndexCountermodels

#check consumed_bridge_does_not_imply_produced_bridge
#check consumed_bridge_rejects_valid_successor

end tests.NebulaFreshClaimIndexCountermodels
