import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.AcceptedApplicationFor

/-! Regression surface for the row-derived recursive F-prime continuation. -/

set_option autoImplicit false

namespace tests.NebulaProductionPaperRecursiveAcceptedApplicationFor

open Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor

#check Rows.statementIdExact
#check Rows.nifsOutputAlias
#check Rows.currentMemoryHeadersPlaced
#check Rows.currentMemory
#check Rows.currentMemoryStartParsed
#check Application.outgoing
#check Application.outgoingParsed
#check Application.successorRows
#check Application.successorPlaced
#check Application.authorityPlaced
#check Application.exactInvocation

end tests.NebulaProductionPaperRecursiveAcceptedApplicationFor
