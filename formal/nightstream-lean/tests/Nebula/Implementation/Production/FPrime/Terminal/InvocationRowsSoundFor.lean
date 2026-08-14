import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.InvocationRowsSoundFor

/-! Regression surface for the exponent-indexed terminal F-prime branch. -/

set_option autoImplicit false

namespace tests.NebulaProductionPaperTerminalInvocationRowsSoundFor

open Nightstream.Implementation.Nebula.ProductionPaperTerminalInvocationRowsSoundFor

#check finalRunning
#check children_stage
#check ProductOpening.coreHolds
#check ProductOpening.holds
#check ExactInvocation.trailingClaimExact
#check ExactInvocation.consumesTrailing
#check exact
#check exactOfHolds

end tests.NebulaProductionPaperTerminalInvocationRowsSoundFor
