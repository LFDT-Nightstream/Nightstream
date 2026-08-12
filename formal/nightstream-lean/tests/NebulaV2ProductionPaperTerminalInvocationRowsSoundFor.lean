import Nightstream.Implementation.NebulaV2.ProductionPaperTerminalInvocationRowsSoundFor

/-! Regression surface for the exponent-indexed terminal F-prime branch. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionPaperTerminalInvocationRowsSoundFor

open Nightstream.Implementation.NebulaV2.ProductionPaperTerminalInvocationRowsSoundFor

#check finalRunning
#check children_stage
#check ProductOpening.coreHolds
#check ProductOpening.holds
#check ExactInvocation.trailingClaimExact
#check ExactInvocation.consumesTrailing
#check exact
#check exactOfHolds

end tests.NebulaV2ProductionPaperTerminalInvocationRowsSoundFor
