import Nightstream.Implementation.NebulaV2.Production.FPrime.Terminal.PublicRowsFor

/-! Surface gate for exact terminal public-result rows. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionPaperTerminalPublicRowsFor

open Nightstream.Implementation.NebulaV2.ProductionPaperTerminalPublicRowsFor

#check rows_length_exact
#check checks_canonical
#check rows_imply_publicChecks

end tests.NebulaV2ProductionPaperTerminalPublicRowsFor
