import Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor

/-! Regression surface for producer-derived same-witness commitment authority
over the complete delayed F-prime lifetime. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionPaperClaimOpeningLifetimeFor

open Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor

#check ClaimLifetime.statementOfClaim
#check ClaimLifetime.statementOfProtocolClaim
#check ClaimLifetime.statementOfProtocolClaim_toProtocolClaim
#check ClaimLifetime.ClaimOpeningWitness
#check ClaimLifetime.ClaimOpening
#check ClaimLifetime.ClaimOpening.bundleOpens
#check ClaimLifetime.ClaimOpening.publicInputExact
#check ClaimLifetime.ClaimOpening.relationSatisfied
#check ClaimLifetime.ClaimOpening.affineOne
#check ClaimLifetime.BaseNode.claimOpening
#check ClaimLifetime.RecursiveNode.nextClaimOpening
#check ClaimLifetime.ReceiptOpening
#check ClaimLifetime.ReceiptOpening.claimHolds
#check ClaimLifetime.ReceiptOpening.bundleOpens
#check ClaimLifetime.ReceiptOpening.publicInputExact
#check ClaimLifetime.ReceiptOpening.relationSatisfied
#check ClaimLifetime.ReceiptOpening.componentOpens
#check ClaimLifetime.ReceiptOpening.affineOne
#check ClaimLifetime.ReceiptOpening.exactDecodedBranch
#check ClaimLifetime.Schedule.everyClaimOpened
#check ClaimLifetime.Schedule.everyReceiptOpened
#check ClaimLifetime.ExactOpenings
#check ClaimLifetime.Lifetime.exactOpenings
#check ClaimLifetime.Lifetime.everyConsumedBundleOpens
#check ClaimLifetime.Lifetime.everyConsumedClaimHolds
#check SemanticLifetime.LifetimeExtraction.precommitChainWithOpenings

end tests.NebulaV2ProductionPaperClaimOpeningLifetimeFor
