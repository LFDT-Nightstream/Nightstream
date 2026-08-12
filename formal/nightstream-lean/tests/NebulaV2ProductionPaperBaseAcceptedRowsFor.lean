import Nightstream.Implementation.NebulaV2.ProductionPaperBaseAcceptedRowsFor

/-! Regression surface for the row-accepted base F-prime package. -/

namespace tests.NebulaV2ProductionPaperBaseAcceptedRowsFor

open Nightstream.Implementation.NebulaV2.ProductionPaperBaseAcceptedRowsFor

#check Accepted.initialValueExact
#check Accepted.initialClosedExact
#check Accepted.openedExists
#check Accepted.opened
#check Accepted.activeOfWireExact
#check Accepted.outgoingSemanticExact
#check Supplement.memoryResult
#check Supplement.outgoingValue_eq_firstBoundary
#check Supplement.memoryStartsAt
#check Supplement.challengeAuthorityExact
#check Supplement.evidence

end tests.NebulaV2ProductionPaperBaseAcceptedRowsFor
