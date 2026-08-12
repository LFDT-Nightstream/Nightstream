import Nightstream.Implementation.NebulaV2.Production.Carrier.NifsPublicCarrierFor

/-! Regression surface for the exponent-indexed carrier-to-PiCCS bridge. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionFullClaimNifsPublicCarrierFor

open Nightstream.Implementation.NebulaV2.ProductionFullClaimNifsPublicCarrierFor

#check fields_length
#check PrefixCanonical
#check fieldValues_eq_publicNifsFields
#check bindPublicFields
#check RemainingPlacement
#check piCcsPlacement

end tests.NebulaV2ProductionFullClaimNifsPublicCarrierFor
