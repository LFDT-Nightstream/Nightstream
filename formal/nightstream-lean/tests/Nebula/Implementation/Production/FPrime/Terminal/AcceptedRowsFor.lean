import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.AcceptedRowsFor

/-! Regression surface for the row-accepted terminal F-prime package. -/

namespace tests.NebulaProductionPaperTerminalAcceptedRowsFor

open Nightstream.Implementation.Nebula.ProductionPaperTerminalAcceptedRowsFor

#check Rows.existsResult
#check Rows.recursive
#check Rows.recursiveCompactManifestExact
#check Rows.result
#check Accepted.opening
#check Accepted.exactInvocation

end tests.NebulaProductionPaperTerminalAcceptedRowsFor
