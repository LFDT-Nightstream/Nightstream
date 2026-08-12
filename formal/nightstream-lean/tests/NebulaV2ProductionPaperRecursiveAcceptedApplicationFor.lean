import Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedApplicationFor

/-! Regression surface for the row-derived recursive F-prime continuation. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionPaperRecursiveAcceptedApplicationFor

open Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor

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

end tests.NebulaV2ProductionPaperRecursiveAcceptedApplicationFor
