import Nightstream.Implementation.Nebula.Production.FPrime.Base.AcceptedRowsFor

/-! Regression surface for the row-accepted base F-prime package. -/

namespace tests.NebulaProductionPaperBaseAcceptedRowsFor

open Nightstream.Implementation.Nebula.ProductionPaperBaseAcceptedRowsFor

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

end tests.NebulaProductionPaperBaseAcceptedRowsFor
