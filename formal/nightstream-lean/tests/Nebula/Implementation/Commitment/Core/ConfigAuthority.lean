import Nightstream.Implementation.Nebula.Commitment.Core.ConfigAuthority

/-! Surface checks for verifier-owned product-commitment authority. -/

set_option autoImplicit false

namespace tests.NebulaProductCommitmentConfigAuthority

open Nightstream.Implementation.Nebula.ProductCommitmentConfigAuthority

#check Authority.config
#check ofTerminalLayout
#check config_ofTerminalLayout
#check config_lanes
#check config_fullKey
#check config_operationsKey
#check config_snapshotKey

end tests.NebulaProductCommitmentConfigAuthority
