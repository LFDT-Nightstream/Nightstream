import Nightstream.Implementation.Nebula.Production.Carrier.NifsPublicCarrierFor

/-! Regression surface for the exponent-indexed carrier-to-PiCCS bridge. -/

set_option autoImplicit false

namespace tests.NebulaProductionFullClaimNifsPublicCarrierFor

open Nightstream.Implementation.Nebula.ProductionFullClaimNifsPublicCarrierFor

#check fields_length
#check PrefixCanonical
#check fieldValues_eq_publicNifsFields
#check bindPublicFields
#check RemainingPlacement
#check piCcsPlacement

end tests.NebulaProductionFullClaimNifsPublicCarrierFor
