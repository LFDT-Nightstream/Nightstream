import Nightstream.Implementation.Nebula.Production.Carrier.StateBinding

/-! Regression surface for the field-native full-claim state binding. -/

set_option autoImplicit false

namespace tests.NebulaProductionFullClaimStateBinding

open Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding

#check authoritativeFrame_length
#check authoritativeFrame_lengthFor
#check bindingState_replays_authoritativeFrame
#check equal_bindingState_recovers_claim_or_named_failure
#check authoritativeFrames_ne_of_candidate_ne

end tests.NebulaProductionFullClaimStateBinding
