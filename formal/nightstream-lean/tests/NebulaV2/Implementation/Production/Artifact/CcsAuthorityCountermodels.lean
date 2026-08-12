import Nightstream.Implementation.NebulaV2.Production.Artifact.CcsAuthorityCountermodels

/-! Regression surface for incomplete production CCS authority countermodels. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionCcsAuthorityCountermodels

open Nightstream.Implementation.NebulaV2.ProductionCcsAuthorityCountermodels

#check badAffine_memoryMatches
#check badAffine_not_fullMatches
#check wrongState_memoryMatches
#check wrongState_not_fullMatches
#check badPadding_memoryMatches
#check badPadding_not_fullMatches

end tests.NebulaV2ProductionCcsAuthorityCountermodels
