import Nightstream.Implementation.NebulaV2.ProductionPaperTerminalStatementRowsFor

/-! Surface gate for exact terminal public-statement recomposition rows. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionPaperTerminalStatementRowsFor

open Nightstream.Implementation.NebulaV2.ProductionPaperTerminalStatementRowsFor

#check rows_length_exact
#check rows_imply_statementPlaced

end tests.NebulaV2ProductionPaperTerminalStatementRowsFor
